from ... import ability
from ... import classifying
from ...hearing import voice

import abc
import dataclasses
import typing

import hsm
import pydantic

from bot.telemetry import observer

class VoiceIdentificationSegment(pydantic.BaseModel):
    """Segmented voice audio paired with diarization metadata."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {
                    "diarization": {
                        "speaker_label": "speaker_1",
                        "start_seconds": 0.0,
                        "end_seconds": 1.25,
                        "confidence": 0.87,
                    },
                    "audio": "c3BlYWtlciBhdWRpbw==",
                }
            ],
        },
    )

    diarization: voice.diarization.VoiceDiarizationSegment = pydantic.Field(
        description=(
            "Diarization metadata describing the speaker label and time span for the audio segment being identified."
        ),
        examples=[
            {
                "speaker_label": "speaker_1",
                "start_seconds": 0.0,
                "end_seconds": 1.25,
                "confidence": 0.87,
            }
        ],
    )
    audio: bytes = pydantic.Field(
        min_length=1,
        description=(
            "Audio bytes clipped to the diarized speaker segment. The bytes should contain only the voice span "
            "described by the diarization metadata. JSON callers must provide this field as base64url-encoded bytes."
        ),
        examples=["c3BlYWtlciBhdWRpbw=="],
    )

class InputData(pydantic.BaseModel):
    """Segmented voice audio to identify by voice signature."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
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
            ],
        },
    )

    segments: tuple[VoiceIdentificationSegment, ...] = pydantic.Field(
        min_length=1,
        description=(
            "Diarized voice segments with their clipped audio bytes, ordered by ascending segment start time."
        ),
        examples=[
            [
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
        ],
    )

    @pydantic.model_validator(mode="after")
    def validate_segment_order(self) -> typing.Self:
        """Require identification segments to keep diarization order."""

        previous_start_seconds: float | None = None
        for segment in self.segments:
            start_seconds = segment.diarization.start_seconds
            if previous_start_seconds is not None and start_seconds < previous_start_seconds:
                raise ValueError("segments must be ordered by ascending diarization.start_seconds.")
            previous_start_seconds = start_seconds
        return self

class VoiceSignature(pydantic.BaseModel):
    """Provider-neutral voice signature for an identified speaker."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "speaker_label": "speaker_1",
                    "signature": "voiceprint:operator-primary",
                    "confidence": 0.91,
                }
            ],
        },
    )

    speaker_label: str = pydantic.Field(
        min_length=1,
        description="Provider-neutral speaker label from the diarization segment this voice signature identifies.",
        examples=["speaker_1"],
    )
    signature: str = pydantic.Field(
        min_length=1,
        description=(
            "Opaque voice signature, voiceprint key, or embedding reference that can be used to recognize this speaker "
            "in later voice-identification operations."
        ),
        examples=["voiceprint:operator-primary"],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional provider confidence that the voice signature represents this speaker, normalized from 0.0 to 1.0."
        ),
        examples=[0.91],
    )

class OutputData(pydantic.BaseModel):
    """Voice signatures identified from segmented voice audio."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "signatures": [
                        {
                            "speaker_label": "speaker_1",
                            "signature": "voiceprint:operator-primary",
                            "confidence": 0.91,
                        }
                    ]
                }
            ],
        },
    )

    signatures: tuple[VoiceSignature, ...] = pydantic.Field(
        min_length=1,
        description="Voice signatures identified for the diarized speakers in the input audio.",
        examples=[
            [
                {
                    "speaker_label": "speaker_1",
                    "signature": "voiceprint:operator-primary",
                    "confidence": 0.91,
                }
            ]
        ],
    )

    @pydantic.model_validator(mode="after")
    def validate_unique_speaker_labels(self) -> typing.Self:
        """Require each output speaker label to resolve to at most one signature."""

        speaker_labels: set[str] = set()
        for signature in self.signatures:
            if signature.speaker_label in speaker_labels:
                raise ValueError("signatures must identify each speaker_label at most once.")
            speaker_labels.add(signature.speaker_label)
        return self

class VoiceIdentifier(classifying.Classifier[InputData, OutputData], abc.ABC):
    """Classifier that identifies segmented voice audio and returns voice signatures."""

_VoiceIdentificationApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.hearing.voice.identification.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_VoiceIdentificationApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.hearing.voice.identification.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)

def _has_voice_identification_input(
    ctx: hsm.Context,
    instance: "VoiceIdentification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, InputData)

def _has_voice_identification_output(
    ctx: hsm.Context,
    instance: "VoiceIdentification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, OutputData)

def _has_invalid_voice_identification_output(
    ctx: hsm.Context,
    instance: "VoiceIdentification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, OutputData)

def _has_voice_identification_failure(
    ctx: hsm.Context,
    instance: "VoiceIdentification",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)

class VoiceIdentification(classifying.Classifying[InputData, OutputData]):
    """Ability to identify diarized voice segments by voice signature."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = ability.ability_input_event(
        "bot.ability.hearing.voice.identification.input",
        InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = ability.ability_output_event(
        "bot.ability.hearing.voice.identification.output",
        OutputData,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _VoiceIdentificationApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _VoiceIdentificationApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_voice_identification_output_failure(
        ctx: hsm.Context,
        instance: "VoiceIdentification",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(
            message="VoiceIdentification produced output that does not match its output schema."
        )
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "VoiceIdentification",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_voice_identification_input),
                hsm.target("../applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(classifying.Classifying._run_behavior_activity),
            hsm.transition(
                hsm.on(_VoiceIdentificationApplyCompletedEvent),
                hsm.guard(_has_voice_identification_output),
                hsm.effect(classifying.Classifying._dispatch_classifying_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_VoiceIdentificationApplyCompletedEvent),
                hsm.guard(_has_invalid_voice_identification_output),
                hsm.effect(_dispatch_invalid_voice_identification_output_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_VoiceIdentificationApplyFailedEvent),
                hsm.guard(_has_voice_identification_failure),
                hsm.effect(classifying.Classifying._dispatch_classifying_failure),
                hsm.target("../idle"),
            ),
        ),
        hsm.observe(observer),
    )
