from mosfet import abilities
from mosfet.abilities.hearing import voice

import asyncio
import collections.abc
import json
import math
import typing
from typing import override

import hsm
import pytest

from tests.bot.abilities.support import start_abilities_for_test
from tests.type_helpers import object_dict


SEGMENT_AUDIO = b"\x00\x00"


class FixedVoiceClassifier(abilities.Classifier[voice.identification.InputData, voice.identification.OutputData]):
    @override
    async def classify(self, input: voice.identification.InputData) -> voice.identification.OutputData:
        del input
        return voice.identification.OutputData(
            embeddings=(
                voice.identification.VoiceEmbedding(
                    embedding=(0.12, -0.08, 0.31),
                    model="voice-embedding-v1",
                    confidence=0.91,
                ),
            )
        )


class RecordingVoiceIdentification(voice.identification.VoiceIdentification):
    outputs: list[voice.identification.OutputData]

    def __init__(
        self,
        *,
        classifier: abilities.Classifier[voice.identification.InputData, voice.identification.OutputData],
    ) -> None:
        super().__init__(classifier=classifier)
        self.outputs = []

    @override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, voice.identification.OutputData)
            self.outputs.append(output)
        return super().dispatch(ctx, event)


async def await_voice_identification(
    output: collections.abc.Awaitable[voice.identification.OutputData],
) -> voice.identification.OutputData:
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


def test_voice_identification_accepts_provider_neutral_segmented_audio() -> None:
    segment = voice.VoiceSegment(
        audio=SEGMENT_AUDIO,
        media_type="audio/pcm",
        sample_rate_hz=4,
        channels=1,
        start_seconds=0.0,
        end_seconds=0.25,
    )
    input = voice.identification.InputData(segments=(segment,))

    assert input.segments == (segment,)
    assert type(input.segments[0]) is voice.VoiceSegment
    assert input.segments[0].start_seconds == 0.0
    assert input.segments[0].audio == SEGMENT_AUDIO


def test_voice_identification_input_rejects_ambiguous_segments() -> None:
    with pytest.raises(ValueError):
        _ = voice.VoiceDiarizationSegment(
            audio=b"",
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            channels=1,
            start_seconds=0.0,
            end_seconds=1.25,
            confidence=0.87,
        )

    with pytest.raises(ValueError):
        _ = voice.identification.InputData(segments=())

    with pytest.raises(ValueError):
        _ = voice.identification.InputData(
            segments=(
                voice.VoiceSegment(
                    audio=b"\x00\x00" * 3,
                    media_type="audio/pcm",
                    sample_rate_hz=4,
                    channels=1,
                    start_seconds=1.25,
                    end_seconds=2.0,
                ),
                voice.VoiceSegment(
                    audio=b"\x00\x00" * 5,
                    media_type="audio/pcm",
                    sample_rate_hz=4,
                    channels=1,
                    start_seconds=0.0,
                    end_seconds=1.25,
                ),
            )
        )


def test_voice_identification_input_round_trips_segment_audio_as_base64_json() -> None:
    payload = {
        "segments": [
            {
                "audio": "AAA=",
                "media_type": "audio/pcm",
                "sample_rate_hz": 4,
                "channels": 1,
                "start_seconds": 0.0,
                "end_seconds": 0.25,
            }
        ]
    }

    input = voice.identification.InputData.model_validate_json(json.dumps(payload))
    schema = object_dict(voice.identification.InputData.model_json_schema())
    definitions = object_dict(schema["$defs"])
    segment_schema = object_dict(definitions["VoiceSegment"])
    properties = object_dict(segment_schema["properties"])
    audio_property_schema = object_dict(properties["audio"])

    assert input.segments[0].audio == SEGMENT_AUDIO
    assert json.loads(input.model_dump_json())["segments"][0]["audio"] == "AAA="
    assert audio_property_schema["format"] == "base64url"
    assert "confidence" not in properties
    assert "speaker_label" not in properties
    assert "diarization" not in properties


def test_voice_identification_output_records_voice_signature() -> None:
    embedding = voice.identification.VoiceEmbedding(
        embedding=(0.12, -0.08, 0.31),
        model="voice-embedding-v1",
        confidence=0.91,
    )
    output = voice.identification.OutputData(embeddings=(embedding,))

    assert output.embeddings == (embedding,)
    assert output.embeddings[0].embedding == (0.12, -0.08, 0.31)
    assert output.embeddings[0].model == "voice-embedding-v1"
    assert output.embeddings[0].confidence == 0.91


def test_voice_identification_output_rejects_invalid_signatures() -> None:
    with pytest.raises(ValueError):
        _ = voice.identification.VoiceEmbedding(embedding=(), model="voice-embedding-v1", confidence=0.91)

    with pytest.raises(ValueError):
        _ = voice.identification.VoiceEmbedding(embedding=(0.1,), model="", confidence=0.91)

    with pytest.raises(ValueError):
        _ = voice.identification.VoiceEmbedding(embedding=(0.1,), model="voice-embedding-v1", confidence=1.1)

    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            _ = voice.identification.VoiceEmbedding(embedding=(invalid,), model="voice-embedding-v1")

    with pytest.raises(ValueError):
        _ = voice.identification.OutputData(embeddings=())


def test_voice_identification_uses_injected_identifier() -> None:
    async def run() -> list[voice.identification.OutputData]:
        ability = RecordingVoiceIdentification(classifier=FixedVoiceClassifier())
        await start_ability_tree(None, ability)

        _ = await ability.apply(voice.identification.InputData(segments=(_segment(),)))
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs[0].embeddings[0].embedding == (0.12, -0.08, 0.31)


def test_voice_identification_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(voice.identification.VoiceIdentification.input_event.schema)
    output_schema = object_dict(voice.identification.VoiceIdentification.output_event.schema)

    assert voice.identification.VoiceIdentification.input_event.name == "bot.ability.hearing.voice.identification.input"
    assert input_schema == voice.identification.InputData.model_json_schema()
    assert input_schema["description"]

    assert (
        voice.identification.VoiceIdentification.output_event.name == "bot.ability.hearing.voice.identification.output"
    )
    assert output_schema == voice.identification.OutputData.model_json_schema()
    assert output_schema["description"]


def test_voice_identification_is_concrete_ability() -> None:
    ability = voice.identification.VoiceIdentification(classifier=FixedVoiceClassifier())

    assert isinstance(ability, abilities.Ability)
    assert isinstance(ability.classifier, abilities.Classifier)
    assert not hasattr(voice.identification, "VoiceIdentifier")
    assert voice.identification.VoiceIdentification.input_data_type is voice.identification.InputData
    assert voice.identification.VoiceIdentification.output_data_type is voice.identification.OutputData


def _segment() -> voice.VoiceSegment:
    return voice.VoiceSegment(
        audio=SEGMENT_AUDIO,
        media_type="audio/pcm",
        sample_rate_hz=4,
        channels=1,
        start_seconds=0.0,
        end_seconds=0.25,
    )
