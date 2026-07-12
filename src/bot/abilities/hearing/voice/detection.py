from ... import ability
from ... import classifying

import abc
import dataclasses
import typing

import hsm
import pydantic

from bot.telemetry import observer

class OutputData(pydantic.BaseModel):
    """Provider-neutral result of detecting voice presence in hearing input."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {"is_voice": True, "confidence": 0.92},
                {"is_voice": False, "confidence": 0.81},
            ],
        },
    )

    is_voice: bool = pydantic.Field(
        description="Whether the hearing input was detected as containing voice.",
        examples=[True],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional provider confidence for the voice/non-voice decision, normalized from 0.0 to 1.0 when available."
        ),
        examples=[0.92],
    )

class VoiceDetector(classifying.Classifier[bytes, OutputData], abc.ABC):
    """Classifier that detects whether raw hearing input contains voice."""

_VoiceDetectionApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.hearing.voice.detection.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_VoiceDetectionApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.hearing.voice.detection.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)

def _has_voice_detection_input(
    ctx: hsm.Context,
    instance: "VoiceDetection",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, bytes)

def _has_detected_voice(
    ctx: hsm.Context,
    instance: "VoiceDetection",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, OutputData) and data.is_voice

def _has_detected_no_voice(
    ctx: hsm.Context,
    instance: "VoiceDetection",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, OutputData) and not data.is_voice

def _has_invalid_voice_detection_output(
    ctx: hsm.Context,
    instance: "VoiceDetection",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return not isinstance(event.data, OutputData)

def _has_voice_detection_failure(
    ctx: hsm.Context,
    instance: "VoiceDetection",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ability.FailureData)

class VoiceDetection(classifying.Classifying[bytes, OutputData]):
    """Ability to detect whether hearing input contains voice."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = bytes
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[bytes]] = ability.ability_input_event(
        "bot.ability.hearing.voice.detection.input",
        bytes,
        description="Raw hearing audio bytes to inspect for voice presence.",
        examples=["base64-encoded audio bytes"],
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = ability.ability_output_event(
        "bot.ability.hearing.voice.detection.output",
        OutputData,
    )

    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _VoiceDetectionApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _VoiceDetectionApplyFailedEvent

    @staticmethod
    def _dispatch_invalid_voice_detection_output_failure(
        ctx: hsm.Context,
        instance: "VoiceDetection",
        event: hsm.Event[typing.Any],
    ) -> None:
        failure = ability.FailureData(
            message="Voice detection produced output that does not match its output schema."
        )
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "VoiceDetection",
        hsm.initial(hsm.target("NoVoiceDetected")),
        hsm.state(
            "NoVoiceDetected",
            hsm.initial(hsm.target("Monitoring")),
            hsm.state(
                "Monitoring",
                hsm.transition(
                    hsm.on(input_event),
                    hsm.guard(_has_voice_detection_input),
                    hsm.target("../Detecting"),
                ),
            ),
            hsm.state(
                "Detecting",
                hsm.defer(input_event),
                hsm.activity(classifying.Classifying._run_behavior_activity),
                hsm.transition(
                    hsm.on(_VoiceDetectionApplyCompletedEvent),
                    hsm.guard(_has_detected_voice),
                    hsm.effect(classifying.Classifying._dispatch_classifying_output),
                    hsm.target("../../VoiceDetected/Monitoring"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDetectionApplyCompletedEvent),
                    hsm.guard(_has_detected_no_voice),
                    hsm.effect(classifying.Classifying._dispatch_classifying_output),
                    hsm.target("../Monitoring"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDetectionApplyCompletedEvent),
                    hsm.guard(_has_invalid_voice_detection_output),
                    hsm.effect(_dispatch_invalid_voice_detection_output_failure),
                    hsm.target("../Monitoring"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDetectionApplyFailedEvent),
                    hsm.guard(_has_voice_detection_failure),
                    hsm.effect(classifying.Classifying._dispatch_classifying_failure),
                    hsm.target("../Monitoring"),
                ),
            ),
        ),
        hsm.state(
            "VoiceDetected",
            hsm.initial(hsm.target("Monitoring")),
            hsm.state(
                "Monitoring",
                hsm.transition(
                    hsm.on(input_event),
                    hsm.guard(_has_voice_detection_input),
                    hsm.target("../Detecting"),
                ),
            ),
            hsm.state(
                "Detecting",
                hsm.defer(input_event),
                hsm.activity(classifying.Classifying._run_behavior_activity),
                hsm.transition(
                    hsm.on(_VoiceDetectionApplyCompletedEvent),
                    hsm.guard(_has_detected_voice),
                    hsm.effect(classifying.Classifying._dispatch_classifying_output),
                    hsm.target("../Monitoring"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDetectionApplyCompletedEvent),
                    hsm.guard(_has_detected_no_voice),
                    hsm.effect(classifying.Classifying._dispatch_classifying_output),
                    hsm.target("../../NoVoiceDetected/Monitoring"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDetectionApplyCompletedEvent),
                    hsm.guard(_has_invalid_voice_detection_output),
                    hsm.effect(_dispatch_invalid_voice_detection_output_failure),
                    hsm.target("../Monitoring"),
                ),
                hsm.transition(
                    hsm.on(_VoiceDetectionApplyFailedEvent),
                    hsm.guard(_has_voice_detection_failure),
                    hsm.effect(classifying.Classifying._dispatch_classifying_failure),
                    hsm.target("../Monitoring"),
                ),
            ),
        ),
        hsm.observe(observer),
    )
