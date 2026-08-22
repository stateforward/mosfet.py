from ... import ability
from ... import classifying
from . import segment

import abc
import dataclasses
import typing

import hsm
import bot
import pydantic

from bot.telemetry import observer


class InputData(pydantic.BaseModel):
    """Raw PCM audio and the format needed by a voice diarizer."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {
                    "audio": "AAA=",
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 48000,
                    "channels": 1,
                }
            ],
        },
    )

    audio: bytes = pydantic.Field(
        min_length=1,
        description="Raw signed 16-bit PCM audio bytes to diarize; JSON callers provide base64-encoded bytes.",
        examples=["AAAA"],
    )
    media_type: typing.Literal["audio/pcm"] = pydantic.Field(
        description="Media type of audio; diarization accepts raw signed 16-bit PCM only.",
        examples=["audio/pcm"],
    )
    sample_rate_hz: int = pydantic.Field(
        ge=1,
        description="PCM sample rate in hertz.",
        examples=[48000],
    )
    channels: int = pydantic.Field(
        ge=1,
        description="Number of interleaved PCM channels.",
        examples=[1, 2],
    )


class VoiceDiarizationSegment(segment.VoiceSegment):
    """One clipped, time-bounded audio segment produced by voice diarization."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {
                    "audio": "AAA=",
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 4,
                    "channels": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 0.25,
                    "confidence": 0.87,
                }
            ],
        },
    )

    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional provider confidence for this speaker attribution, normalized from 0.0 to 1.0.",
        examples=[0.87],
    )


class OutputData(pydantic.BaseModel):
    """Provider-neutral diarization result for voice input."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "segments": [
                        {
                            "audio": "AAA=",
                            "media_type": "audio/pcm",
                            "sample_rate_hz": 4,
                            "channels": 1,
                            "start_seconds": 0.0,
                            "end_seconds": 0.25,
                            "confidence": 0.87,
                        },
                        {
                            "audio": "AAAAAA==",
                            "media_type": "audio/pcm",
                            "sample_rate_hz": 4,
                            "channels": 1,
                            "start_seconds": 0.25,
                            "end_seconds": 0.75,
                            "confidence": 0.82,
                        },
                    ]
                }
            ],
        },
    )

    segments: tuple[VoiceDiarizationSegment, ...] = pydantic.Field(
        min_length=1,
        description=(
            "Clipped audio segments ordered by ascending start time in the voice input. Each segment "
            "retains its timing and optional diarization confidence."
        ),
        examples=[
            [
                {
                    "audio": "AAA=",
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 4,
                    "channels": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 0.25,
                    "confidence": 0.87,
                }
            ]
        ],
    )

    @pydantic.model_validator(mode="after")
    def validate_segment_order(self) -> typing.Self:
        """Require diarization segments to be ordered by start time."""

        previous_start_seconds: float | None = None
        for voice_segment in self.segments:
            if previous_start_seconds is not None and voice_segment.start_seconds < previous_start_seconds:
                raise ValueError("segments must be ordered by ascending start_seconds.")
            previous_start_seconds = voice_segment.start_seconds
        return self


class VoiceDiarizer(classifying.Classifier[InputData, OutputData], abc.ABC):
    """Classifier that diarizes raw PCM voice input into speaker-attributed segments."""


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
    return isinstance(event.data, InputData)


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


class VoiceDiarization(classifying.Classifying[InputData, OutputData]):
    """Ability to diarize hearing input by speaker."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.hearing.voice.diarization.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.hearing.voice.diarization.output",
        schema=OutputData,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _VoiceDiarizationApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _VoiceDiarizationApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_voice_diarization_output_failure(
        ctx: hsm.Context,
        instance: "VoiceDiarization",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(message="VoiceDiarization produced output that does not match its output schema.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
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
