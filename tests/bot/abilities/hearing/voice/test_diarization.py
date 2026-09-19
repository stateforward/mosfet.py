from mosfet import abilities
from mosfet.abilities.hearing import voice

import asyncio
import collections.abc
import typing
from typing import override

import hsm
import pytest

from tests.bot.abilities.support import start_abilities_for_test
from tests.type_helpers import object_dict


SEGMENT_AUDIO = b"\x00\x00" * 5


class FixedVoiceDiarizer(voice.diarization.VoiceDiarizer):
    @override
    async def classify(self, input: voice.diarization.InputData) -> voice.diarization.OutputData:
        del input
        return voice.diarization.OutputData(
            segments=(
                voice.diarization.VoiceDiarizationSegment(
                    audio=SEGMENT_AUDIO,
                    media_type="audio/pcm",
                    sample_rate_hz=4,
                    channels=1,
                    start_seconds=0.0,
                    end_seconds=1.25,
                    confidence=0.87,
                ),
            )
        )


class RecordingVoiceDiarization(voice.diarization.VoiceDiarization):
    outputs: list[voice.diarization.OutputData]

    def __init__(self, *, classifier: voice.diarization.VoiceDiarizer) -> None:
        super().__init__(classifier=classifier)
        self.outputs = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, voice.diarization.OutputData)
            self.outputs.append(output)
        return super().dispatch(ctx, event)


async def await_voice_diarization(
    output: collections.abc.Awaitable[voice.diarization.OutputData],
) -> voice.diarization.OutputData:
    return await output


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0)


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


async def start_ability_tree(ctx: hsm.Context | None, ability: abilities.Ability[typing.Any, typing.Any]) -> None:
    await start_abilities_for_test(hsm.Context() if ctx is None else ctx, ability)


def test_voice_diarization_output_records_diarized_segments() -> None:
    segment = voice.diarization.VoiceDiarizationSegment(
        audio=SEGMENT_AUDIO,
        media_type="audio/pcm",
        sample_rate_hz=4,
        channels=1,
        start_seconds=0.0,
        end_seconds=1.25,
        confidence=0.87,
    )
    diarization = voice.diarization.OutputData(segments=(segment,))

    assert diarization.segments == (segment,)
    assert diarization.segments[0].audio == SEGMENT_AUDIO
    assert diarization.segments[0].start_seconds == 0.0
    assert diarization.segments[0].end_seconds == 1.25
    assert diarization.segments[0].confidence == 0.87


def test_voice_diarization_segment_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        _ = voice.diarization.VoiceDiarizationSegment(
            audio=SEGMENT_AUDIO,
            media_type="audio/pcm",
            sample_rate_hz=4,
            channels=1,
            start_seconds=-0.1,
            end_seconds=1.25,
            confidence=0.87,
        )

    with pytest.raises(ValueError):
        _ = voice.diarization.VoiceDiarizationSegment(
            audio=SEGMENT_AUDIO,
            media_type="audio/pcm",
            sample_rate_hz=4,
            channels=1,
            start_seconds=1.25,
            end_seconds=1.0,
            confidence=0.87,
        )

    with pytest.raises(ValueError):
        _ = voice.diarization.VoiceDiarizationSegment(
            audio=SEGMENT_AUDIO,
            media_type="audio/pcm",
            sample_rate_hz=4,
            channels=1,
            start_seconds=0.0,
            end_seconds=1.25,
            confidence=1.1,
        )


def test_voice_diarization_output_rejects_ambiguous_segments() -> None:
    first = voice.diarization.VoiceDiarizationSegment(
        audio=b"\x00\x00",
        media_type="audio/pcm",
        sample_rate_hz=4,
        channels=1,
        start_seconds=1.0,
        end_seconds=1.25,
        confidence=0.87,
    )
    second = voice.diarization.VoiceDiarizationSegment(
        audio=b"\x00\x00" * 3,
        media_type="audio/pcm",
        sample_rate_hz=4,
        channels=1,
        start_seconds=0.0,
        end_seconds=0.75,
        confidence=0.81,
    )

    with pytest.raises(ValueError):
        _ = voice.diarization.VoiceDiarizationSegment(
            audio=b"",
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            channels=1,
            start_seconds=0.0,
            end_seconds=1.25,
            confidence=0.87,
        )

    with pytest.raises(ValueError):
        _ = voice.diarization.OutputData(segments=())

    with pytest.raises(ValueError):
        _ = voice.diarization.OutputData(segments=(first, second))


def test_voice_diarization_segment_rejects_malformed_pcm_duration() -> None:
    with pytest.raises(ValueError, match="complete 16-bit PCM frames"):
        _ = voice.diarization.VoiceDiarizationSegment(
            audio=b"\x00",
            media_type="audio/pcm",
            sample_rate_hz=4,
            channels=1,
            start_seconds=0.0,
            end_seconds=0.25,
        )

    with pytest.raises(ValueError, match="audio duration"):
        _ = voice.diarization.VoiceDiarizationSegment(
            audio=b"\x00\x00",
            media_type="audio/pcm",
            sample_rate_hz=4,
            channels=1,
            start_seconds=0.0,
            end_seconds=1.0,
        )

    rounded = voice.diarization.VoiceDiarizationSegment(
        audio=b"\x00\x00" * 3,
        media_type="audio/pcm",
        sample_rate_hz=10,
        channels=1,
        start_seconds=0.0,
        end_seconds=0.28,
    )
    assert rounded.duration_seconds == pytest.approx(0.3)


def test_voice_diarization_uses_injected_diarizer() -> None:
    async def run() -> list[voice.diarization.OutputData]:
        ability = RecordingVoiceDiarization(classifier=FixedVoiceDiarizer())
        await start_ability_tree(None, ability)

        _ = await ability.apply(
            voice.diarization.InputData(
                audio=b"audio",
                media_type="audio/pcm",
                sample_rate_hz=48_000,
                channels=1,
            )
        )
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs[0].segments[0].audio == SEGMENT_AUDIO


def test_voice_diarization_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(voice.diarization.VoiceDiarization.input_event.schema)
    output_schema = object_dict(voice.diarization.VoiceDiarization.output_event.schema)

    assert voice.diarization.VoiceDiarization.input_event.name == "bot.ability.hearing.voice.diarization.input"
    assert input_schema == voice.diarization.InputData.model_json_schema()
    assert input_schema["description"]

    assert voice.diarization.VoiceDiarization.output_event.name == "bot.ability.hearing.voice.diarization.output"
    assert output_schema == voice.diarization.OutputData.model_json_schema()
    assert output_schema["description"]


def test_voice_diarization_is_concrete_ability() -> None:
    ability = voice.diarization.VoiceDiarization(classifier=FixedVoiceDiarizer())

    assert isinstance(ability, abilities.Ability)
    assert issubclass(voice.diarization.VoiceDiarizer, abilities.Classifier)
    assert voice.diarization.VoiceDiarization.output_data_type is voice.diarization.OutputData
