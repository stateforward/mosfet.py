from bot import abilities
from bot.abilities.hearing import voice

import asyncio
import collections.abc
import json
import typing
from typing import override

import hsm
import pytest

from tests.bot.abilities.support import start_abilities_for_test
from tests.type_helpers import object_dict

class FixedVoiceIdentifier(voice.identification.VoiceIdentifier):
    @override
    async def classify(self, input: voice.identification.InputData) -> voice.identification.OutputData:
        del input
        return voice.identification.OutputData(
            signatures=(
                voice.identification.VoiceSignature(
                    speaker_label="speaker_1",
                    signature="voiceprint:operator-primary",
                    confidence=0.91,
                ),
            )
        )

class RecordingVoiceIdentification(voice.identification.VoiceIdentification):
    outputs: list[voice.identification.OutputData]

    def __init__(self, *, classifier: voice.identification.VoiceIdentifier) -> None:
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

def test_voice_identification_accepts_segmented_audio() -> None:
    diarized_segment = voice.VoiceDiarizationSegment(
        speaker_label="speaker_1",
        start_seconds=0.0,
        end_seconds=1.25,
        confidence=0.87,
    )
    segment = voice.identification.VoiceIdentificationSegment(
        diarization=diarized_segment,
        audio=b"speaker audio",
    )
    input = voice.identification.InputData(segments=(segment,))

    assert input.segments == (segment,)
    assert input.segments[0].diarization.speaker_label == "speaker_1"
    assert input.segments[0].audio == b"speaker audio"

def test_voice_identification_input_rejects_ambiguous_segments() -> None:
    diarized_segment = voice.VoiceDiarizationSegment(
        speaker_label="speaker_1",
        start_seconds=0.0,
        end_seconds=1.25,
        confidence=0.87,
    )

    with pytest.raises(ValueError):
        _ = voice.identification.VoiceIdentificationSegment(diarization=diarized_segment, audio=b"")

    with pytest.raises(ValueError):
        _ = voice.identification.InputData(segments=())

    with pytest.raises(ValueError):
        _ = voice.identification.InputData(
            segments=(
                voice.identification.VoiceIdentificationSegment(
                    diarization=voice.VoiceDiarizationSegment(
                        speaker_label="speaker_1",
                        start_seconds=1.25,
                        end_seconds=2.0,
                        confidence=0.87,
                    ),
                    audio=b"second",
                ),
                voice.identification.VoiceIdentificationSegment(
                    diarization=voice.VoiceDiarizationSegment(
                        speaker_label="speaker_1",
                        start_seconds=0.0,
                        end_seconds=1.25,
                        confidence=0.87,
                    ),
                    audio=b"first",
                ),
            )
        )

def test_voice_identification_input_round_trips_segment_audio_as_base64_json() -> None:
    payload = {
        "segments": [
            {
                "diarization": {
                    "speaker_label": "speaker_1",
                    "start_seconds": 0.0,
                    "end_seconds": 1.25,
                    "confidence": 0.87,
                },
                "audio": "c3BlYWtlciBhdWRpbw==",
            }
        ]
    }

    input = voice.identification.InputData.model_validate_json(json.dumps(payload))
    schema = object_dict(voice.identification.InputData.model_json_schema())
    definitions = object_dict(schema["$defs"])
    segment_schema = object_dict(definitions["VoiceIdentificationSegment"])
    properties = object_dict(segment_schema["properties"])
    audio_property_schema = object_dict(properties["audio"])

    assert input.segments[0].audio == b"speaker audio"
    assert json.loads(input.model_dump_json())["segments"][0]["audio"] == "c3BlYWtlciBhdWRpbw=="
    assert audio_property_schema["format"] == "base64url"

def test_voice_identification_output_records_voice_signature() -> None:
    signature = voice.identification.VoiceSignature(
        speaker_label="speaker_1",
        signature="voiceprint:operator-primary",
        confidence=0.91,
    )
    output = voice.identification.OutputData(signatures=(signature,))

    assert output.signatures == (signature,)
    assert output.signatures[0].speaker_label == "speaker_1"
    assert output.signatures[0].signature == "voiceprint:operator-primary"
    assert output.signatures[0].confidence == 0.91

def test_voice_identification_output_rejects_invalid_signatures() -> None:
    with pytest.raises(ValueError):
        _ = voice.identification.VoiceSignature(speaker_label="", signature="voiceprint:operator-primary", confidence=0.91)

    with pytest.raises(ValueError):
        _ = voice.identification.VoiceSignature(speaker_label="speaker_1", signature="", confidence=0.91)

    with pytest.raises(ValueError):
        _ = voice.identification.VoiceSignature(speaker_label="speaker_1", signature="voiceprint:operator-primary", confidence=1.1)

    with pytest.raises(ValueError):
        _ = voice.identification.OutputData(signatures=())

    with pytest.raises(ValueError):
        _ = voice.identification.OutputData(
            signatures=(
                voice.identification.VoiceSignature(speaker_label="speaker_1", signature="voiceprint:operator-primary", confidence=0.91),
                voice.identification.VoiceSignature(speaker_label="speaker_1", signature="voiceprint:operator-secondary", confidence=0.88),
            )
        )

def test_voice_identification_uses_injected_identifier() -> None:
    async def run() -> list[voice.identification.OutputData]:
        ability = RecordingVoiceIdentification(classifier=FixedVoiceIdentifier())
        await start_ability_tree(None, ability)

        _ = await ability.apply(voice.identification.InputData(segments=(_segment(),)))
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs[0].signatures[0].signature == "voiceprint:operator-primary"

def test_voice_identification_events_use_concrete_pydantic_schemas() -> None:
    input_schema = object_dict(voice.identification.VoiceIdentification.input_event.schema)
    output_schema = object_dict(voice.identification.VoiceIdentification.output_event.schema)

    assert voice.identification.VoiceIdentification.input_event.name == "bot.ability.hearing.voice.identification.input"
    assert input_schema == voice.identification.InputData.model_json_schema()
    assert input_schema["description"]

    assert voice.identification.VoiceIdentification.output_event.name == "bot.ability.hearing.voice.identification.output"
    assert output_schema == voice.identification.OutputData.model_json_schema()
    assert output_schema["description"]

def test_voice_identification_is_concrete_ability() -> None:
    ability = voice.identification.VoiceIdentification(classifier=FixedVoiceIdentifier())

    assert isinstance(ability, abilities.Ability)
    assert issubclass(voice.identification.VoiceIdentifier, abilities.Classifier)
    assert voice.identification.VoiceIdentification.input_data_type is voice.identification.InputData
    assert voice.identification.VoiceIdentification.output_data_type is voice.identification.OutputData

def _segment() -> voice.identification.VoiceIdentificationSegment:
    return voice.identification.VoiceIdentificationSegment(
        diarization=voice.VoiceDiarizationSegment(
            speaker_label="speaker_1",
            start_seconds=0.0,
            end_seconds=1.25,
            confidence=0.87,
        ),
        audio=b"speaker audio",
    )
