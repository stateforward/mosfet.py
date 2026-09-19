from ... import ability
from ... import classifying
from . import segment

import dataclasses
import math
import typing

import hsm
import mosfet
import pydantic


class InputData(pydantic.BaseModel):
    """Provider-neutral voice segments to identify by voice embedding."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {
                    "segments": [
                        {
                            "audio": "c3BlYWtlciBhdWRpbw==",
                            "media_type": "audio/pcm",
                            "sample_rate_hz": 16000,
                            "channels": 1,
                            "start_seconds": 0.0,
                            "end_seconds": 1.25,
                        }
                    ]
                }
            ],
        },
    )

    segments: tuple[segment.VoiceSegment, ...] = pydantic.Field(
        min_length=1,
        description=("Provider-neutral voice segments with clipped audio bytes, ordered by ascending start time."),
        examples=[
            [
                {
                    "audio": "c3BlYWtlciBhdWRpbw==",
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 1.25,
                }
            ]
        ],
    )

    @pydantic.model_validator(mode="after")
    def validate_segment_order(self) -> typing.Self:
        """Require identification segments to be ordered by start time."""

        previous_start_seconds: float | None = None
        for voice_segment in self.segments:
            start_seconds = voice_segment.start_seconds
            if previous_start_seconds is not None and start_seconds < previous_start_seconds:
                raise ValueError("segments must be ordered by ascending start_seconds.")
            previous_start_seconds = start_seconds
        return self


class VoiceEmbedding(pydantic.BaseModel):
    """Provider-neutral embedding value returned for an identified voice segment."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "embedding": [0.12, -0.08, 0.31],
                    "model": "voice-embedding-v1",
                    "confidence": 0.91,
                }
            ],
        },
    )

    embedding: tuple[float, ...] = pydantic.Field(
        min_length=1,
        description=(
            "Provider-neutral numeric voice embedding produced by the classifier. Providers decide how to "
            "index and compare the vector; it is not a human-readable speaker label."
        ),
        examples=[[0.12, -0.08, 0.31]],
    )
    model: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional embedding model identifier that produced this voice embedding.",
        examples=["voice-embedding-v1"],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional provider confidence that the voice embedding represents this speaker, normalized from 0.0 to 1.0."
        ),
        examples=[0.91],
    )

    @pydantic.field_validator("embedding")
    @classmethod
    def validate_finite_embedding(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Reject NaN and infinite vector components at the provider-neutral boundary."""

        if any(not math.isfinite(component) for component in value):
            raise ValueError("embedding values must be finite.")
        return value


class OutputData(pydantic.BaseModel):
    """Voice embeddings identified from segmented voice audio."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "embeddings": [
                        {
                            "embedding": [0.12, -0.08, 0.31],
                            "model": "voice-embedding-v1",
                            "confidence": 0.91,
                        }
                    ]
                }
            ],
        },
    )

    embeddings: tuple[VoiceEmbedding, ...] = pydantic.Field(
        min_length=1,
        description="Voice embeddings identified for the input voice segments, in input order.",
        examples=[
            [
                {
                    "embedding": [0.12, -0.08, 0.31],
                    "model": "voice-embedding-v1",
                    "confidence": 0.91,
                }
            ]
        ],
    )


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
    """Ability to identify segmented voice audio by voice embedding."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.hearing.voice.identification.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.hearing.voice.identification.output",
        schema=OutputData,
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
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
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
    )
