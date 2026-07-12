from ... import ability
from ... import classifying

import abc
import dataclasses
import typing

import hsm
import pydantic

from bot.telemetry import observer

class VoiceDiarizationSegment(pydantic.BaseModel):
    """A speaker-attributed time span in diarized voice input."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "speaker_label": "speaker_1",
                    "start_seconds": 0.0,
                    "end_seconds": 1.25,
                    "confidence": 0.87,
                }
            ],
        },
    )

    speaker_label: str = pydantic.Field(
        min_length=1,
        description="Provider-neutral speaker label that is stable within this diarization result.",
        examples=["speaker_1"],
    )
    start_seconds: float = pydantic.Field(
        ge=0.0,
        description="Inclusive start time of this speaker segment, measured in seconds from the beginning of input.",
        examples=[0.0],
    )
    end_seconds: float = pydantic.Field(
        ge=0.0,
        description="Exclusive end time of this speaker segment, measured in seconds from the beginning of input.",
        examples=[1.25],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional provider confidence for this speaker attribution, normalized from 0.0 to 1.0.",
        examples=[0.87],
    )

    @pydantic.model_validator(mode="after")
    def validate_time_span(self) -> typing.Self:
        """Require each diarized segment to cover a positive time span."""

        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds.")
        return self

class OutputData(pydantic.BaseModel):
    """Provider-neutral diarization result for voice input."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "segments": [
                        {
                            "speaker_label": "speaker_1",
                            "start_seconds": 0.0,
                            "end_seconds": 1.25,
                            "confidence": 0.87,
                        },
                        {
                            "speaker_label": "speaker_2",
                            "start_seconds": 1.25,
                            "end_seconds": 2.0,
                            "confidence": 0.82,
                        },
                    ]
                }
            ],
        },
    )

    segments: tuple[VoiceDiarizationSegment, ...] = pydantic.Field(
        min_length=1,
        description="Speaker-attributed segments ordered by ascending start time in the voice input.",
        examples=[
            [
                {
                    "speaker_label": "speaker_1",
                    "start_seconds": 0.0,
                    "end_seconds": 1.25,
                    "confidence": 0.87,
                }
            ]
        ],
    )

    @pydantic.model_validator(mode="after")
    def validate_segment_order(self) -> typing.Self:
        """Require diarization segments to be ordered by start time."""

        previous_start_seconds: float | None = None
        for segment in self.segments:
            if previous_start_seconds is not None and segment.start_seconds < previous_start_seconds:
                raise ValueError("segments must be ordered by ascending start_seconds.")
            previous_start_seconds = segment.start_seconds
        return self

class VoiceDiarizer(classifying.Classifier[bytes, OutputData], abc.ABC):
    """Classifier that diarizes raw voice input into speaker-attributed segments."""

_VoiceDiarizationApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.hearing.voice.diarization.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_VoiceDiarizationApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.hearing.voice.diarization.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)

def _has_voice_diarization_input(
    ctx: hsm.Context,
    instance: "VoiceDiarization",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, bytes)

def _has_voice_diarization_output(
    ctx: hsm.Context,
    instance: "VoiceDiarization",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, OutputData)

def _has_invalid_voice_diarization_output(
    ctx: hsm.Context,
    instance: "VoiceDiarization",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, OutputData)

def _has_voice_diarization_failure(
    ctx: hsm.Context,
    instance: "VoiceDiarization",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)

class VoiceDiarization(classifying.Classifying[bytes, OutputData]):
    """Ability to diarize hearing input by speaker."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = bytes
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[bytes]] = ability.ability_input_event(
        "bot.ability.hearing.voice.diarization.input",
        bytes,
        description="Raw hearing audio bytes to diarize by speaker.",
        examples=["base64-encoded audio bytes"],
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = ability.ability_output_event(
        "bot.ability.hearing.voice.diarization.output",
        OutputData,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _VoiceDiarizationApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _VoiceDiarizationApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_voice_diarization_output_failure(
        ctx: hsm.Context,
        instance: "VoiceDiarization",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(
            message="VoiceDiarization produced output that does not match its output schema."
        )
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "VoiceDiarization",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_voice_diarization_input),
                hsm.target("../applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(classifying.Classifying._run_behavior_activity),
            hsm.transition(
                hsm.on(_VoiceDiarizationApplyCompletedEvent),
                hsm.guard(_has_voice_diarization_output),
                hsm.effect(classifying.Classifying._dispatch_classifying_output),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_VoiceDiarizationApplyCompletedEvent),
                hsm.guard(_has_invalid_voice_diarization_output),
                hsm.effect(_dispatch_invalid_voice_diarization_output_failure),
                hsm.target("../idle"),
            ),
            hsm.transition(
                hsm.on(_VoiceDiarizationApplyFailedEvent),
                hsm.guard(_has_voice_diarization_failure),
                hsm.effect(classifying.Classifying._dispatch_classifying_failure),
                hsm.target("../idle"),
            ),
        ),
        hsm.observe(observer),
    )
