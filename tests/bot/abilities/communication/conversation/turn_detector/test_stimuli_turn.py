"""Direct contract tests for turn_detector stimuli and turn-boundary events."""

from mosfet import media
from mosfet.abilities.communication.conversation import turn_detector

import pydantic
import pytest

_PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"png-body", media_type="image/png")


def test_text_stimulus_carries_participant_content() -> None:
    stimulus = turn_detector.TextStimulus(source_participant_ref="caller", content="hello")

    assert stimulus.kind == "text"
    assert stimulus.source_participant_ref == "caller"
    assert stimulus.content == "hello"


def test_audio_stimulus_defaults_to_audio_kind() -> None:
    stimulus = turn_detector.AudioStimulus(source_participant_ref="caller", content=b"pcm-bytes")

    assert stimulus.kind == "audio"
    assert stimulus.content == b"pcm-bytes"


def test_event_stimulus_defaults_to_empty_payload() -> None:
    stimulus = turn_detector.EventStimulus(source_participant_ref="caller", event="custom.happened")

    assert stimulus.kind == "event"
    assert stimulus.payload == {}


def test_turn_start_accepts_matching_first_chunk() -> None:
    start = turn_detector.TurnStartData(
        turn_ref="turn-1",
        content=turn_detector.TextStimulus(source_participant_ref="caller", content="hello"),
    )

    assert start.turn_ref == "turn-1"
    assert start.source_participant_ref == "caller"
    assert turn_detector.TurnStartEvent.name == "bot.ability.turn_detector.turn.start"


def test_turn_boundaries_reject_content_from_another_participant() -> None:
    with pytest.raises(ValueError, match="source_participant_ref"):
        _ = turn_detector.TurnStartData(
            turn_ref="turn-1",
            source_participant_ref="caller",
            content=turn_detector.TextStimulus(source_participant_ref="bot", content="hello"),
        )
    with pytest.raises(ValueError, match="source_participant_ref"):
        _ = turn_detector.TurnEndData(
            turn_ref="turn-1",
            source_participant_ref="caller",
            content=turn_detector.TextStimulus(source_participant_ref="bot", content="bye"),
        )


def test_image_stimulus_carries_a_typed_image() -> None:
    stimulus = turn_detector.ImageStimulus(source_participant_ref="caller", content=_PNG)

    assert stimulus.kind == "image"
    assert stimulus.content == _PNG
    assert stimulus.content.media_type == "image/png"


def test_image_stimulus_rejects_raw_bytes() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = turn_detector.ImageStimulus.model_validate(
            {"source_participant_ref": "caller", "content": b"\x89PNG\r\n\x1a\nraw"}
        )


def test_turn_update_accepts_an_image_chunk() -> None:
    update = turn_detector.TurnUpdateData(
        turn_ref="turn-1",
        content=turn_detector.ImageStimulus(source_participant_ref="caller", content=_PNG),
    )

    assert isinstance(update.content, turn_detector.ImageStimulus)
    assert update.content.content == _PNG


def test_turn_update_requires_a_text_or_audio_chunk() -> None:
    update = turn_detector.TurnUpdateData(
        turn_ref="turn-1",
        content=turn_detector.TextStimulus(source_participant_ref="caller", content="world"),
    )

    assert update.turn_ref == "turn-1"
    assert turn_detector.TurnUpdateEvent.name == "bot.ability.turn_detector.turn.update"
    assert turn_detector.TurnPauseEvent.name == "bot.ability.turn_detector.turn.pause"
    assert turn_detector.TurnEndEvent.name == "bot.ability.turn_detector.turn.end"
