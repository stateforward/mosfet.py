"""Direct contract tests for the trusted Communication SpeechHeard seed.

The seed is native HSM (not learned inventory): it wires an identified Listening
product into a typed Communication input. These tests pin the descriptor contract
and the admit/reject behavior through public ability seams only.
"""

from mosfet.abilities import cognition
from mosfet.abilities import communication
from mosfet.abilities import listening
from mosfet.abilities import processing
from mosfet.abilities.communication import behaviors
from mosfet.abilities.communication import conversation
from mosfet.abilities.hearing import voice

import asyncio
import typing

from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test


def _speech(*, source_ids: typing.Any, content: str = "hello") -> listening.SpeechData:
    return listening.SpeechData(
        content=content,
        content_type="text/plain",
        voice_detection=voice.detection.ApplyData(
            segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=0.1, confidence=0.9),)
        ),
        sample_rate_hz=16_000,
        media_type="audio/pcm",
        channels=1,
        source_ids=source_ids,
    )


def _seed_input(speech: listening.SpeechData) -> cognition.InputData:
    return cognition.InputData(
        stimulus=listening.SpeechEvent.with_data(speech),
        abilities=(),
        actors={},
        focus=None,
        focus_candidates=(),
    )


def test_speech_heard_seed_descriptor_is_stable() -> None:
    seed = communication.speech_heard_seed()

    assert seed.name == behaviors.SPEECH_HEARD_NAME == "SpeechHeard"
    assert seed.triggers == behaviors.SPEECH_HEARD_TRIGGERS == (listening.SpeechEvent.name,)
    first = seed.factory()
    second = seed.factory()
    assert isinstance(first, communication.SpeechHeard)
    assert isinstance(second, communication.SpeechHeard)
    assert first is not second
    adapted_input = _seed_input(_speech(source_ids=frozenset({(0.12, -0.08, 0.31)})))
    assert seed.input_adapter is not None
    assert seed.input_adapter(adapted_input) is adapted_input


def test_speech_heard_admits_identified_speech_into_communication() -> None:
    async def run() -> processing.OutputData:
        seed = communication.speech_heard_seed()
        heard = typing.cast(communication.SpeechHeard, seed.factory())
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, heard)
        speech = _speech(source_ids=frozenset({(0.12, -0.08, 0.31)}), content="answer the phone")
        return await dispatch_ability_for_test(heard, ctx, _seed_input(speech))

    output = asyncio.run(run())

    assert output.handled
    assert len(output.events) == 1
    selected = output.events[0]
    assert selected.event == communication.InputEvent.name
    assert isinstance(selected.data, conversation.TurnData)
    assert selected.data.source_ids == frozenset({(0.12, -0.08, 0.31)})
    assert selected.data.content == "answer the phone"
    assert selected.data.content_type == "text/plain"
    assert selected.data.sample_rate_hz == 16_000
    assert selected.data.channels == 1


def test_speech_heard_declines_speech_without_source_identity() -> None:
    async def run() -> processing.OutputData:
        seed = communication.speech_heard_seed()
        heard = typing.cast(communication.SpeechHeard, seed.factory())
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, heard)
        return await dispatch_ability_for_test(heard, ctx, _seed_input(_speech(source_ids=frozenset())))

    output = asyncio.run(run())

    assert output.handled is False
    assert output.events == ()
