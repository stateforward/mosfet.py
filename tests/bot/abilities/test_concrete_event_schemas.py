from bot.abilities.hearing import speech as hearing_speech
from bot.abilities.vocal import speech as vocal_speech

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

def test_speech_encoding_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(vocal_speech.encoding.SpeechEncoding.input_event.schema)
    output_schema = object_dict(vocal_speech.encoding.SpeechEncoding.output_event.schema)
    expected_input_schema = _type_schema(bytes)
    expected_output_schema = _type_schema(bytes)

    assert vocal_speech.encoding.SpeechEncoding.input_event.name == "bot.ability.vocal.speech.encoding.input"
    assert input_schema["type"] == expected_input_schema["type"]
    assert input_schema["format"] == expected_input_schema["format"]

    assert vocal_speech.encoding.SpeechEncoding.output_event.name == "bot.ability.vocal.speech.encoding.output"
    assert output_schema["type"] == expected_output_schema["type"]
    assert output_schema["format"] == expected_output_schema["format"]
