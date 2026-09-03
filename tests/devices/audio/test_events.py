from bot.devices import audio

from tests.type_helpers import object_dict


def test_audio_events_use_pydantic_schemas() -> None:
    input_schema = object_dict(audio.InputEvent.schema)
    output_schema = object_dict(audio.OutputEvent.schema)

    assert audio.InputEvent.name == "devices.audio.input"
    assert input_schema == audio.InputData.model_json_schema()
    assert input_schema["required"] == ["audio"]
    assert audio.OutputEvent.name == "devices.audio.output"
    assert output_schema == audio.OutputData.model_json_schema()
    assert output_schema["required"] == ["audio"]


def test_audio_event_data_describes_generic_sound_chunks() -> None:
    input_data = audio.InputData(audio=b"captured-audio", media_type="audio/opus", sample_rate_hz=48_000, channels=1)
    output_data = audio.OutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=44_100, channels=2)

    assert input_data.audio == b"captured-audio"
    assert input_data.media_type == "audio/opus"
    assert input_data.sample_rate_hz == 48_000
    assert input_data.channels == 1
    assert output_data.audio == b"playback-audio"
    assert output_data.media_type == "audio/pcm"
    assert output_data.sample_rate_hz == 44_100
    assert output_data.channels == 2


def test_audio_event_data_rejects_empty_audio() -> None:
    try:
        _ = audio.InputData(audio=b"")
    except ValueError:
        pass
    else:
        raise AssertionError("Audio input data should reject empty audio chunks.")

    try:
        _ = audio.OutputData(audio=b"")
    except ValueError:
        pass
    else:
        raise AssertionError("Audio output data should reject empty audio chunks.")
