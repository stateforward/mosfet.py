from mosfet.devices import audio

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


def test_speaker_control_events_use_pydantic_schemas() -> None:
    assert (audio.FlushEvent.name, audio.FlushEvent.schema) == ("devices.audio.flush", audio.FlushData)
    assert (audio.PlayedEvent.name, audio.PlayedEvent.schema) == ("devices.audio.played", audio.PlayedData)
    assert (audio.FlushedEvent.name, audio.FlushedEvent.schema) == ("devices.audio.flushed", audio.FlushedData)
    assert object_dict(audio.PlayedData.model_json_schema())["required"] == ["mark"]
    assert audio.FlushedData(dropped_seconds=0.2, marks=("u1.1", "u1.2")).marks == ("u1.1", "u1.2")


def test_output_mark_is_an_optional_bounded_label() -> None:
    assert audio.OutputData(audio=b"\x00\x00").mark is None
    assert audio.OutputData(audio=b"\x00\x00", mark="u1.1").mark == "u1.1"
    for mark in ("", "m" * 65):
        try:
            _ = audio.OutputData(audio=b"\x00\x00", mark=mark)
        except ValueError:
            continue
        raise AssertionError(f"OutputData should reject the mark {mark!r}.")


def test_speaker_reports_reject_negative_durations_and_empty_marks() -> None:
    for build in (
        lambda: audio.FlushedData(dropped_seconds=-0.1, marks=()),
        lambda: audio.FlushedData(dropped_seconds=0.1, marks=(), played_seconds=-0.1),
        lambda: audio.PlayedData(mark=""),
    ):
        try:
            _ = build()
        except ValueError:
            continue
        raise AssertionError("Speaker reports should validate their fields.")
