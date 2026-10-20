from mosfet.abilities import memory
from mosfet.abilities.hearing import speech as hearing_speech

import pydantic

from tests.type_helpers import object_dict


def _type_schema(data_type: type[object]) -> dict[str, object]:
    return object_dict(pydantic.TypeAdapter(data_type).json_schema())


def test_speech_decoding_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(hearing_speech.decoding.SpeechDecoding.input_event.schema)
    output_schema = object_dict(hearing_speech.decoding.SpeechDecoding.output_event.schema)
    expected_input_schema = _type_schema(bytes)
    expected_output_schema = _type_schema(bytes)

    assert hearing_speech.decoding.SpeechDecoding.input_event.name == "bot.ability.hearing.speech.decoding.input"
    assert input_schema["type"] == expected_input_schema["type"]
    assert input_schema["format"] == expected_input_schema["format"]
    # Bare ``bytes`` schema — no event-level examples ceremony.

    assert hearing_speech.decoding.SpeechDecoding.output_event.name == "bot.ability.hearing.speech.decoding.output"
    assert output_schema["type"] == expected_output_schema["type"]
    assert output_schema["format"] == expected_output_schema["format"]


def test_knowing_people_events_use_concrete_pydantic_schemas() -> None:
    """R-12: what the bot learns about people is memory's capacity, under memory's event names."""

    events = (
        (memory.KnowEvent, memory.KnowData, "bot.ability.memory.know"),
        (memory.ForgetEvent, memory.ForgetData, "bot.ability.memory.forget"),
        (memory.KnownEvent, memory.KnownData, "bot.ability.memory.known"),
    )

    for event, data, name in events:
        assert event.name == name
        assert object_dict(event.schema) == object_dict(data.model_json_schema())
